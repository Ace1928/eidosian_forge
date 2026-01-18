from ._base import *
import operator as op
@classmethod
def get_hash_schema(cls):
    return {'password': 'hash_password'}