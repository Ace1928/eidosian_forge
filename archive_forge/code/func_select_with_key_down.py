from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def select_with_key_down(self, keyboard, scancode, codepoint, modifiers, **kwargs):
    """Processes a key press. This is called when a key press is to be used
        for selection. Depending on the keyboard keys pressed and the
        configuration, it could select or deselect nodes or node ranges
        from the selectable nodes list, :meth:`get_selectable_nodes`.

        The parameters are such that it could be bound directly to the
        on_key_down event of a keyboard. Therefore, it is safe to be called
        repeatedly when the key is held down as is done by the keyboard.

        :Returns:
            bool, True if the keypress was used, False otherwise.
        """
    if not self.keyboard_select:
        return False
    keys = self._key_list
    multi = self.multiselect
    node_src, idx_src = self._resolve_last_node()
    text = scancode[1]
    if text == 'shift':
        self._shift_down = True
    elif text in ('ctrl', 'lctrl', 'rctrl'):
        self._ctrl_down = True
    elif multi and 'ctrl' in modifiers and (text in ('a', 'A')) and (text not in keys):
        sister_nodes = self.get_selectable_nodes()
        select = self.select_node
        for node in sister_nodes:
            select(node)
        keys.append(text)
    else:
        s = text
        if len(text) > 1:
            d = {'divide': '/', 'mul': '*', 'substract': '-', 'add': '+', 'decimal': '.'}
            if text.startswith('numpad'):
                s = text[6:]
                if len(s) > 1:
                    if s in d:
                        s = d[s]
                    else:
                        s = None
            else:
                s = None
        if s is not None:
            if s not in keys:
                if time() - self._last_key_time <= self.text_entry_timeout:
                    self._word_filter += s
                else:
                    self._word_filter = s
                keys.append(s)
            self._last_key_time = time()
            node, idx = self.goto_node(self._word_filter, node_src, idx_src)
        else:
            self._word_filter = ''
            node, idx = self.goto_node(text, node_src, idx_src)
        if node == node_src:
            return False
        multiselect = multi and 'ctrl' in modifiers
        if multi and 'shift' in modifiers:
            self._select_range(multiselect, True, node, idx)
        else:
            if not multiselect:
                self.clear_selection()
            self.select_node(node)
        return True
    self._word_filter = ''
    return False