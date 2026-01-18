from parlai.core.teachers import (
from parlai.core.opt import Opt
import copy
import random
import itertools
import os
from PIL import Image
import string
import json
from abc import ABC
from typing import Tuple, List
def image_id_to_image_path(self, image_id):
    """
        Return path to image on disk.
        """
    return os.path.join(self.opt['datapath'], 'ImageTeacher/images', f'{image_id}.jpg')