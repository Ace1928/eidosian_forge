def dump_trans(self, key, set, file):
    file.write('      %s --> %s\n' % (key, self.dump_set(set)))