from kivy.properties import StringProperty
def delete_word_right(self):
    """Delete text right of the cursor to the end of the word"""
    if self._selection:
        return
    start_index = self.cursor_index()
    start_cursor = self.cursor
    self.do_cursor_movement('cursor_right', control=True)
    end_index = self.cursor_index()
    if start_index != end_index:
        s = self.text[start_index:end_index]
        self._set_unredo_delsel(start_index, end_index, s, from_undo=False)
        self.text = self.text[:start_index] + self.text[end_index:]
        self._set_cursor(pos=start_cursor)