import os, subprocess, json
@property
def commit_count(self):
    """Return the number of commits since the last release"""
    return self.fetch()._commit_count