import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
@staticmethod
def set_bundle(md, value):
    md.bundle = value