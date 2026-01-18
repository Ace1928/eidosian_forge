from __future__ import annotations
import json
from os import makedirs
from os.path import exists, expanduser
from pymatgen.analysis.chemenv.utils.scripts_utils import strategies_class_lookup
from pymatgen.core import SETTINGS
@property
def has_materials_project_access(self):
    """
        Whether MP access is enabled.
        """
    return self.materials_project_configuration is not None