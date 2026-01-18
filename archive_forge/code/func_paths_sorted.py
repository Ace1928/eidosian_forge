import os
def paths_sorted(self):
    paths = sorted(self.paths.items(), key=lambda key_value: len(key_value[0]), reversed=True)