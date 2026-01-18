from abc import ABC
import inspect
import hashlib
def call_job_fn(self, key, job_fn, args, context):
    raise NotImplementedError