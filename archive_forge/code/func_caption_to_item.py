import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def caption_to_item(self, caption):
    """Get a MenuItem from the caption.

    Args:
      caption: (str) The caption to look up.

    Returns:
      (MenuItem) The first-match menu item with the caption, if any.

    Raises:
      LookupError: If a menu item with the caption does not exist.
    """
    captions = self.captions()
    if caption not in captions:
        raise LookupError('There is no menu item with the caption "%s"' % caption)
    return self._items[captions.index(caption)]