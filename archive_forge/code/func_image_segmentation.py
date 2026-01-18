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
def image_segmentation(self, image: ContentT, *, model: Optional[str]=None) -> List[ImageSegmentationOutput]:
    """
        Perform image segmentation on the given image using the specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The image to segment. It can be raw bytes, an image file, or a URL to an online image.
            model (`str`, *optional*):
                The model to use for image segmentation. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for image segmentation will be used.

        Returns:
            `List[Dict]`: A list of dictionaries containing the segmented masks and associated attributes.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.image_segmentation("cat.jpg"):
        [{'score': 0.989008, 'label': 'LABEL_184', 'mask': <PIL.PngImagePlugin.PngImageFile image mode=L size=400x300 at 0x7FDD2B129CC0>}, ...]
        ```
        """
    response = self.post(data=image, model=model, task='image-segmentation')
    output = _bytes_to_dict(response)
    if not isinstance(output, list):
        raise ValueError(f'Server output must be a list. Got {type(output)}: {str(output)[:200]}...')
    for item in output:
        item['mask'] = _b64_to_image(item['mask'])
    return output