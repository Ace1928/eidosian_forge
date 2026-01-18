from hashlib import md5
import array
import re
def old_db_hash(mfld):
    return md5(old_combined_hash(mfld)).hexdigest()