def stemWord(self, word):
    self.set_current(word)
    self._stem()
    return self.get_current()