import statemachine
def press_button(self, button):
    if button in 'ABCD':
        self._pressed = button
        self.press_alpha_button()
    elif button in '1234':
        self._pressed = button
        self.press_digit_button()
    else:
        print('Did not recognize button {!r}'.format(str(button)))