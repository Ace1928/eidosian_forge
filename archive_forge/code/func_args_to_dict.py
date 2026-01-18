import time
def args_to_dict(args, attrs):
    return {(attr, value) for attr, value in [(attr, getattr(args, attr)) for attr in attrs] if value is not None}