import atexit
import builtins
import inspect
import keyword
import re
import __main__
def global_matches(self, text):
    """Compute matches when text is a simple name.

        Return a list of all keywords, built-in functions and names currently
        defined in self.namespace that match.

        """
    matches = []
    seen = {'__builtins__'}
    n = len(text)
    for word in keyword.kwlist + keyword.softkwlist:
        if word[:n] == text:
            seen.add(word)
            if word in {'finally', 'try'}:
                word = word + ':'
            elif word not in {'False', 'None', 'True', 'break', 'continue', 'pass', 'else', '_'}:
                word = word + ' '
            matches.append(word)
    for nspace in [self.namespace, builtins.__dict__]:
        for word, val in nspace.items():
            if word[:n] == text and word not in seen:
                seen.add(word)
                matches.append(self._callable_postfix(val, word))
    return matches