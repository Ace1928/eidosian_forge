import base64
import copy
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from io import BytesIO
from xml.dom import minidom
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
def set_preference(self, key, value):
    """Sets the preference that we want in the profile."""
    self._desired_preferences[key] = value