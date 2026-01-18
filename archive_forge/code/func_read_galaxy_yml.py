from __future__ import annotations
import json
import os
import re
import sys
import yaml
def read_galaxy_yml(collection_path):
    """Return collection information from the galaxy.yml file."""
    galaxy_path = os.path.join(collection_path, 'galaxy.yml')
    if not os.path.exists(galaxy_path):
        return None
    try:
        with open(galaxy_path, encoding='utf-8') as galaxy_file:
            galaxy = yaml.safe_load(galaxy_file)
        result = dict(version=galaxy.get('version'))
        validate_version(result['version'])
    except Exception as ex:
        raise Exception('{0}: {1}'.format(os.path.basename(galaxy_path), ex)) from None
    return result