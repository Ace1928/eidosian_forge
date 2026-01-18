import statemachine
def press_digit_button(self):
    try:
        super(VendingMachine, self).press_digit_button()
    except VendingMachineState.InvalidTransitionException as ite:
        print(ite)
    else:
        self._digit_pressed = self._pressed
        self.dispense()