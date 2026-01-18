from kivy.properties import StringProperty
def keyboard_on_key_down(self, window, keycode, text, modifiers):
    key, key_str = keycode
    mod = '+'.join(modifiers) if modifiers else None
    is_emacs_shortcut = False
    if key in range(256) and self.key_bindings == 'emacs':
        if mod == 'ctrl' and chr(key) in self.bindings['ctrl'].keys():
            is_emacs_shortcut = True
        elif mod == 'alt' and chr(key) in self.bindings['alt'].keys():
            is_emacs_shortcut = True
        else:
            is_emacs_shortcut = False
    if is_emacs_shortcut:
        emacs_shortcut = self.bindings[mod][chr(key)]
        emacs_shortcut()
    else:
        super(EmacsBehavior, self).keyboard_on_key_down(window, keycode, text, modifiers)