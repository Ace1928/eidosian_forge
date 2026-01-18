import math
from typing import Literal, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from torchmetrics.functional.image.lpips import _LPIPS
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE
def perceptual_path_length(generator: GeneratorType, num_samples: int=10000, conditional: bool=False, batch_size: int=64, interpolation_method: Literal['lerp', 'slerp_any', 'slerp_unit']='lerp', epsilon: float=0.0001, resize: Optional[int]=64, lower_discard: Optional[float]=0.01, upper_discard: Optional[float]=0.99, sim_net: Union[nn.Module, Literal['alex', 'vgg', 'squeeze']]='vgg', device: Union[str, torch.device]='cpu') -> Tuple[Tensor, Tensor, Tensor]:
    """Computes the perceptual path length (`PPL`_) of a generator model.

    The perceptual path length can be used to measure the consistency of interpolation in latent-space models. It is
    defined as

    .. math::
        PPL = \\mathbb{E}\\left[\\frac{1}{\\epsilon^2} D(G(I(z_1, z_2, t)), G(I(z_1, z_2, t+\\epsilon)))\\right]

    where :math:`G` is the generator, :math:`I` is the interpolation function, :math:`D` is a similarity metric,
    :math:`z_1` and :math:`z_2` are two sets of latent points, and :math:`t` is a parameter between 0 and 1. The metric
    thus works by interpolating between two sets of latent points, and measuring the similarity between the generated
    images. The expectation is approximated by sampling :math:`z_1` and :math:`z_2` from the generator, and averaging
    the calculated distanced. The similarity metric :math:`D` is by default the `LPIPS`_ metric, but can be changed by
    setting the `sim_net` argument.

    The provided generator model must have a `sample` method with signature `sample(num_samples: int) -> Tensor` where
    the returned tensor has shape `(num_samples, z_size)`. If the generator is conditional, it must also have a
    `num_classes` attribute. The `forward` method of the generator must have signature `forward(z: Tensor) -> Tensor`
    if `conditional=False`, and `forward(z: Tensor, labels: Tensor) -> Tensor` if `conditional=True`. The returned
    tensor should have shape `(num_samples, C, H, W)` and be scaled to the range [0, 255].

    Args:
        generator: Generator model, with specific requirements. See above.
        num_samples: Number of samples to use for the PPL computation.
        conditional: Whether the generator is conditional or not (i.e. whether it takes labels as input).
        batch_size: Batch size to use for the PPL computation.
        interpolation_method: Interpolation method to use. Choose from 'lerp', 'slerp_any', 'slerp_unit'.
        epsilon: Spacing between the points on the path between latent points.
        resize: Resize images to this size before computing the similarity between generated images.
        lower_discard: Lower quantile to discard from the distances, before computing the mean and standard deviation.
        upper_discard: Upper quantile to discard from the distances, before computing the mean and standard deviation.
        sim_net: Similarity network to use. Can be a `nn.Module` or one of 'alex', 'vgg', 'squeeze', where the three
            latter options correspond to the pretrained networks from the `LPIPS`_ paper.
        device: Device to use for the computation.

    Returns:
        A tuple containing the mean, standard deviation and all distances.

    Example::
        >>> from torchmetrics.functional.image import perceptual_path_length
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> class DummyGenerator(torch.nn.Module):
        ...    def __init__(self, z_size) -> None:
        ...       super().__init__()
        ...       self.z_size = z_size
        ...       self.model = torch.nn.Sequential(torch.nn.Linear(z_size, 3*128*128), torch.nn.Sigmoid())
        ...    def forward(self, z):
        ...       return 255 * (self.model(z).reshape(-1, 3, 128, 128) + 1)
        ...    def sample(self, num_samples):
        ...      return torch.randn(num_samples, self.z_size)
        >>> generator = DummyGenerator(2)
        >>> perceptual_path_length(generator, num_samples=10)  # doctest: +SKIP
        (tensor(0.1945),
        tensor(0.1222),
        tensor([0.0990, 0.4173, 0.1628, 0.3573, 0.1875, 0.0335, 0.1095, 0.1887, 0.1953]))

    """
    if not _TORCHVISION_AVAILABLE:
        raise ModuleNotFoundError('Metric `perceptual_path_length` requires torchvision which is not installed.Install with `pip install torchvision` or `pip install torchmetrics[image]`')
    _perceptual_path_length_validate_arguments(num_samples, conditional, batch_size, interpolation_method, epsilon, resize, lower_discard, upper_discard)
    _validate_generator_model(generator, conditional)
    generator = generator.to(device)
    latent1 = generator.sample(num_samples).to(device)
    latent2 = generator.sample(num_samples).to(device)
    latent2 = _interpolate(latent1, latent2, epsilon, interpolation_method=interpolation_method)
    if conditional:
        labels = torch.randint(0, generator.num_classes, (num_samples,)).to(device)
    if isinstance(sim_net, nn.Module):
        net = sim_net.to(device)
    elif sim_net in ['alex', 'vgg', 'squeeze']:
        net = _LPIPS(pretrained=True, net=sim_net, resize=resize).to(device)
    else:
        raise ValueError(f"sim_net must be a nn.Module or one of 'alex', 'vgg', 'squeeze', got {sim_net}")
    with torch.inference_mode():
        distances = []
        num_batches = math.ceil(num_samples / batch_size)
        for batch_idx in range(num_batches):
            batch_latent1 = latent1[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            batch_latent2 = latent2[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            if conditional:
                batch_labels = labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
                outputs = generator(torch.cat((batch_latent1, batch_latent2), dim=0), torch.cat((batch_labels, batch_labels), dim=0))
            else:
                outputs = generator(torch.cat((batch_latent1, batch_latent2), dim=0))
            out1, out2 = outputs.chunk(2, dim=0)
            out1_rescale = 2 * (out1 / 255) - 1
            out2_rescale = 2 * (out2 / 255) - 1
            similarity = net(out1_rescale, out2_rescale)
            dist = similarity / epsilon ** 2
            distances.append(dist.detach())
        distances = torch.cat(distances)
        lower = torch.quantile(distances, lower_discard, interpolation='lower') if lower_discard is not None else 0.0
        upper = torch.quantile(distances, upper_discard, interpolation='lower') if upper_discard is not None else max(distances)
        distances = distances[(distances >= lower) & (distances <= upper)]
        return (distances.mean(), distances.std(), distances)