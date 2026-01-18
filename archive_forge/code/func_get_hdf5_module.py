import gc
import io
import random
import re
import string
import tempfile
from os import environ as env
import h5py
import netCDF4
import numpy as np
import pytest
from packaging import version
from pytest import raises
import h5netcdf
from h5netcdf import legacyapi
from h5netcdf.core import NOT_A_VARIABLE, CompatibilityError
def get_hdf5_module(resource):
    """Return the correct h5py module based on the input resource."""
    if isinstance(resource, str) and resource.startswith(remote_h5):
        return h5pyd
    else:
        return h5py