import gtk
def setSpellChecker(self, checker):
    assert checker, "checker can't be None"
    self._checker = checker
    self._dict_lable.set_text('Dictionary:%s' % (checker.dict.tag,))