import statemachine
def dispense(self):
    try:
        super(VendingMachine, self).dispense()
    except VendingMachineState.InvalidTransitionException as ite:
        print(ite)
    else:
        print('Dispensing at {}{}'.format(self._alpha_pressed, self._digit_pressed))
        self._alpha_pressed = self._digit_pressed = None