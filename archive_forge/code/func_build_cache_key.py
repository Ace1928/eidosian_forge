from abc import ABC
import inspect
import hashlib
def build_cache_key(self, fn, args, cache_args_to_ignore):
    fn_source = inspect.getsource(fn)
    if not isinstance(cache_args_to_ignore, (list, tuple)):
        cache_args_to_ignore = [cache_args_to_ignore]
    if cache_args_to_ignore:
        if isinstance(args, dict):
            args = {k: v for k, v in args.items() if k not in cache_args_to_ignore}
        else:
            args = [arg for i, arg in enumerate(args) if i not in cache_args_to_ignore]
    hash_dict = dict(args=args, fn_source=fn_source)
    if self.cache_by is not None:
        for i, cache_item in enumerate(self.cache_by):
            hash_dict[f'cache_key_{i}'] = cache_item()
    return hashlib.sha1(str(hash_dict).encode('utf-8')).hexdigest()