import base64
import logging
import time
import warnings
from dataclasses import asdict
from typing import (
from requests import HTTPError
from requests.structures import CaseInsensitiveDict
from huggingface_hub.constants import ALL_INFERENCE_API_FRAMEWORKS, INFERENCE_ENDPOINT, MAIN_INFERENCE_API_FRAMEWORKS
from huggingface_hub.inference._common import (
from huggingface_hub.inference._text_generation import (
from huggingface_hub.inference._types import (
from huggingface_hub.utils import (
def image_to_image(self, image: ContentT, prompt: Optional[str]=None, *, negative_prompt: Optional[str]=None, height: Optional[int]=None, width: Optional[int]=None, num_inference_steps: Optional[int]=None, guidance_scale: Optional[float]=None, model: Optional[str]=None, **kwargs) -> 'Image':
    """
        Perform image-to-image translation using a specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The input image for translation. It can be raw bytes, an image file, or a URL to an online image.
            prompt (`str`, *optional*):
                The text prompt to guide the image generation.
            negative_prompt (`str`, *optional*):
                A negative prompt to guide the translation process.
            height (`int`, *optional*):
                The height in pixels of the generated image.
            width (`int`, *optional*):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*):
                Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `Image`: The translated image.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> image = client.image_to_image("cat.jpg", prompt="turn the cat into a tiger")
        >>> image.save("tiger.jpg")
        ```
        """
    parameters = {'prompt': prompt, 'negative_prompt': negative_prompt, 'height': height, 'width': width, 'num_inference_steps': num_inference_steps, 'guidance_scale': guidance_scale, **kwargs}
    if all((parameter is None for parameter in parameters.values())):
        data = image
        payload: Optional[Dict[str, Any]] = None
    else:
        data = None
        payload = {'inputs': _b64_encode(image)}
        for key, value in parameters.items():
            if value is not None:
                payload.setdefault('parameters', {})[key] = value
    response = self.post(json=payload, data=data, model=model, task='image-to-image')
    return _bytes_to_image(response)