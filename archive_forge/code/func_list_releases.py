import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def list_releases():
    """
     Returns dict of Debian releases
    """
    releases = {}
    rels = (('buzz', '1.1'), ('rex', '1.2'), ('bo', '1.3'), ('hamm', '2.0'), ('slink', '2.1'), ('potato', '2.2'), ('woody', '3.0'), ('sarge', '3.1'), ('etch', '4.0'), ('lenny', '5.0'), ('squeeze', '6.0'), ('wheezy', '7'), ('jessie', '8'), ('stretch', '9'), ('buster', '10'), ('bullseye', '11'), ('bookworm', '12'), ('trixie', '13'), ('forky', '14'), ('sid', ''))
    for idx, rel in enumerate(rels):
        name, version = rel
        releases[name] = Release(name, idx, version)
    Release.releases = releases
    return releases