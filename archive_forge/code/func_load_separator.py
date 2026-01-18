from typing import Optional, Union
import torch
import os
import numpy as np
import torchaudio
import warnings
from pathlib import Path
from contextlib import redirect_stderr
import io
import json
import openunmix
from openunmix import model
def load_separator(model_str_or_path: str='umxl', targets: Optional[list]=None, niter: int=1, residual: bool=False, wiener_win_len: Optional[int]=300, device: Union[str, torch.device]='cpu', pretrained: bool=True, filterbank: str='torch'):
    """Separator loader

    Args:
        model_str_or_path (str): Model name or path to model _parent_ directory
            E.g. The following files are assumed to present when
            loading `model_str_or_path='mymodel', targets=['vocals']`
            'mymodel/separator.json', mymodel/vocals.pth', 'mymodel/vocals.json'.
            Defaults to `umxl`.
        targets (list of str or None): list of target names. When loading a
            pre-trained model, all `targets` can be None as all targets
            will be loaded
        niter (int): Number of EM steps for refining initial estimates
            in a post-processing stage. `--niter 0` skips this step altogether
            (and thus makes separation significantly faster) More iterations
            can get better interference reduction at the price of artifacts.
            Defaults to `1`.
        residual (bool): Computes a residual target, for custom separation
            scenarios when not all targets are available (at the expense
            of slightly less performance). E.g vocal/accompaniment
            Defaults to `False`.
        wiener_win_len (int): The size of the excerpts (number of frames) on
            which to apply filtering independently. This means assuming
            time varying stereo models and localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
            Defaults to `300`
        device (str): torch device, defaults to `cpu`
        pretrained (bool): determines if loading pre-trained weights
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """
    model_path = Path(model_str_or_path).expanduser()
    if model_path.exists():
        if targets is None:
            raise UserWarning('For custom models, please specify the targets')
        target_models = load_target_models(targets=targets, model_str_or_path=model_path, pretrained=pretrained)
        with open(Path(model_path, 'separator.json'), 'r') as stream:
            enc_conf = json.load(stream)
        separator = model.Separator(target_models=target_models, niter=niter, residual=residual, wiener_win_len=wiener_win_len, sample_rate=enc_conf['sample_rate'], n_fft=enc_conf['nfft'], n_hop=enc_conf['nhop'], nb_channels=enc_conf['nb_channels'], filterbank=filterbank).to(device)
    else:
        hub_loader = getattr(openunmix, model_str_or_path)
        separator = hub_loader(targets=targets, device=device, pretrained=True, niter=niter, residual=residual, wiener_win_len=wiener_win_len, filterbank=filterbank)
    return separator