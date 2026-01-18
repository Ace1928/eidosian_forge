import os
def get_ctime_macos(filepath):
    return os.stat(filepath).st_birthtime