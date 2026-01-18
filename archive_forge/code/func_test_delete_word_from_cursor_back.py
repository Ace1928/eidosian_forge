import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_delete_word_from_cursor_back(self):
    self.try_stages_kill(['asd;fljk asd;lfjas;dlkfj asdlk jasdf;ljk|', 'asd;fljk asd;lfjas;dlkfj asdlk jasdf;|', 'asd;fljk asd;lfjas;dlkfj asdlk |', 'asd;fljk asd;lfjas;dlkfj |', 'asd;fljk asd;lfjas;|', 'asd;fljk asd;|', 'asd;fljk |', 'asd;|', '|', '|'], delete_word_from_cursor_back)
    self.try_stages_kill([' (( asdf |', ' (( |', '|'], delete_word_from_cursor_back)