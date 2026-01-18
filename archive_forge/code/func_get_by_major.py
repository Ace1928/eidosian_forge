import json
import requests
def get_by_major(version):
    if version.startswith('v'):
        version = version[1:]
    return (version[0], version, int(version.replace('.', '')))