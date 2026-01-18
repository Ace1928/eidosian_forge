import string, sys
def completenames(self, text, *ignored):
    dotext = 'do_' + text
    return [a[3:] for a in self.get_names() if a.startswith(dotext)]