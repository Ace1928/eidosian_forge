import bisect
from _pydevd_bundle.pydevd_constants import NULL, KeyifyList
import pydevd_file_utils
def map_to_client(self, runtime_source_filename, lineno):
    key = (lineno, 'client', runtime_source_filename)
    try:
        return self._cache[key]
    except KeyError:
        for _, mapping in list(self._mappings_to_server.items()):
            for map_entry in mapping:
                if map_entry.runtime_source == runtime_source_filename:
                    if map_entry.contains_runtime_line(lineno):
                        self._cache[key] = (map_entry.source_filename, map_entry.line + (lineno - map_entry.runtime_line), True)
                        return self._cache[key]
        self._cache[key] = (runtime_source_filename, lineno, False)
        return self._cache[key]