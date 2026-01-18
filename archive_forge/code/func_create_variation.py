from __future__ import annotations
from typing import Union, Mapping, Optional, cast
from typing_extensions import Literal
import httpx
from .. import _legacy_response
from ..types import (
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven, FileTypes
from .._utils import (
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from .._base_client import (
def create_variation(self, *, image: FileTypes, model: Union[str, Literal['dall-e-2'], None] | NotGiven=NOT_GIVEN, n: Optional[int] | NotGiven=NOT_GIVEN, response_format: Optional[Literal['url', 'b64_json']] | NotGiven=NOT_GIVEN, size: Optional[Literal['256x256', '512x512', '1024x1024']] | NotGiven=NOT_GIVEN, user: str | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> ImagesResponse:
    """
        Creates a variation of a given image.

        Args:
          image: The image to use as the basis for the variation(s). Must be a valid PNG file,
              less than 4MB, and square.

          model: The model to use for image generation. Only `dall-e-2` is supported at this
              time.

          n: The number of images to generate. Must be between 1 and 10. For `dall-e-3`, only
              `n=1` is supported.

          response_format: The format in which the generated images are returned. Must be one of `url` or
              `b64_json`. URLs are only valid for 60 minutes after the image has been
              generated.

          size: The size of the generated images. Must be one of `256x256`, `512x512`, or
              `1024x1024`.

          user: A unique identifier representing your end-user, which can help OpenAI to monitor
              and detect abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
    body = deepcopy_minimal({'image': image, 'model': model, 'n': n, 'response_format': response_format, 'size': size, 'user': user})
    files = extract_files(cast(Mapping[str, object], body), paths=[['image']])
    if files:
        extra_headers = {'Content-Type': 'multipart/form-data', **(extra_headers or {})}
    return self._post('/images/variations', body=maybe_transform(body, image_create_variation_params.ImageCreateVariationParams), files=files, options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=ImagesResponse)