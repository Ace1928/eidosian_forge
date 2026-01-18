import collections
def bump_version_if_changed(self, name, source_files, build_arguments, build_directory, with_cuda, is_python_module, is_standalone):
    hash_value = 0
    hash_value = hash_source_files(hash_value, source_files)
    hash_value = hash_build_arguments(hash_value, build_arguments)
    hash_value = update_hash(hash_value, build_directory)
    hash_value = update_hash(hash_value, with_cuda)
    hash_value = update_hash(hash_value, is_python_module)
    hash_value = update_hash(hash_value, is_standalone)
    entry = self.entries.get(name)
    if entry is None:
        self.entries[name] = entry = Entry(0, hash_value)
    elif hash_value != entry.hash:
        self.entries[name] = entry = Entry(entry.version + 1, hash_value)
    return entry.version