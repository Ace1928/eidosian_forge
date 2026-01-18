import math
import numpy as np
import pytest
from shapely import (
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry, EmptyGeometry
def empty_generator():
    return iter([])