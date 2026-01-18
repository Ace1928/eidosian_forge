import pickle
import re
from debian.deprecation import function_deprecated_by
def read_tag_database_both_ways(input_data, tag_filter=None):
    """Read the tag database, returning a pkg->tags and a tag->pkgs dictionary"""
    db = {}
    dbr = {}
    for pkgs, tags in parse_tags(input_data):
        if tag_filter is None:
            tags = set(tags)
        else:
            tags = set(filter(tag_filter, tags))
        for pkg in pkgs:
            db[pkg] = tags.copy()
        for tag in tags:
            if tag in dbr:
                dbr[tag] |= pkgs
            else:
                dbr[tag] = pkgs.copy()
    return (db, dbr)