import inspect
import os
import numpy as np
import pytest
import sklearn.datasets
def is_pillow_installed():
    try:
        import PIL
        return True
    except ImportError:
        return False