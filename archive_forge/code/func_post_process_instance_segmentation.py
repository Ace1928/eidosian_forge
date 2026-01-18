from typing import List
from ...processing_utils import ProcessorMixin
from ...utils import is_torch_available
def post_process_instance_segmentation(self, *args, **kwargs):
    """
        This method forwards all its arguments to [`OneFormerImageProcessor.post_process_instance_segmentation`].
        Please refer to the docstring of this method for more information.
        """
    return self.image_processor.post_process_instance_segmentation(*args, **kwargs)