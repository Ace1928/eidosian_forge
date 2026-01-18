import json
import re
import logging
def save_failed_tests(self, result, all_tests):
    print('Saving failed tests to {}'.format(self.cache_filename))
    cache = []
    failed = set()
    for case in result.errors + result.failures:
        failed.add(case[0].id())
    for t in all_tests:
        if t in failed:
            cache.append(t)
    with open(self.cache_filename, 'w') as fout:
        json.dump(cache, fout)