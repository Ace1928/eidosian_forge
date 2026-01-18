import hashlib
import os
import pathlib
import re
import shutil
import tarfile
import urllib
import warnings
import zipfile
from urllib.request import urlretrieve
from keras.src.api_export import keras_export
from keras.src.backend import config
from keras.src.utils import io_utils
from keras.src.utils.module_utils import gfile
from keras.src.utils.progbar import Progbar
def is_link_in_dir(info, base):
    tip = resolve_path(os.path.join(base, os.path.dirname(info.name)))
    return is_path_in_dir(info.linkname, base_dir=tip)