import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def create_attribute_methods(self, obj_attributes):
    for attr in obj_attributes:
        self.__setattr__('set_' + attr, lambda x, a=attr: self.obj_dict['attributes'].__setitem__(a, x))
        self.__setattr__('get_' + attr, lambda a=attr: self.__get_attribute__(a))