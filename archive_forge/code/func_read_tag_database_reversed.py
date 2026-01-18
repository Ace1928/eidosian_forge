import pickle
import re
from debian.deprecation import function_deprecated_by
def read_tag_database_reversed(input_data):
    """Read the tag database, returning a tag->pkgs dictionary"""
    db = {}
    for pkgs, tags in parse_tags(input_data):
        for tag in tags:
            if tag in db:
                db[tag] |= pkgs
            else:
                db[tag] = pkgs.copy()
    return db