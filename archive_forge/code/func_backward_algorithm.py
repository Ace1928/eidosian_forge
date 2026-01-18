import warnings
from Bio import BiopythonDeprecationWarning
def backward_algorithm(self):
    """Calculate sequence probability using the backward algorithm.

        This implements the backward algorithm, as described on p58-59 of
        Durbin et al.

        Returns:
         - A dictionary containing the backwards variables. This has keys
           of the form (state letter, position in the training sequence),
           and values containing the calculated backward variable.

        """
    state_letters = self._mm.state_alphabet
    backward_var = {}
    first_letter = state_letters[0]
    for state in state_letters:
        backward_var[state, len(self._seq.emissions) - 1] = self._mm.transition_prob[state, state_letters[0]]
    all_indexes = list(range(len(self._seq.emissions) - 1))
    all_indexes.reverse()
    for i in all_indexes:
        for main_state in state_letters:
            backward_value = self._backward_recursion(main_state, i, backward_var)
            if backward_value is not None:
                backward_var[main_state, i] = backward_value
    return backward_var