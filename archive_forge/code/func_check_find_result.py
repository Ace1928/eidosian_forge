def check_find_result(self, interval):
    if set(self.find(interval)) != set(self.brute_force_find(interval)):
        raise Exception('Different results: %r %r' % (self.find(interval), self.brute_force_find(interval)))