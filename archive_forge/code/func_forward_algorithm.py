import warnings
from Bio import BiopythonDeprecationWarning
def forward_algorithm(self):
    """Calculate sequence probability using the forward algorithm.

        This implements the forward algorithm, as described on p57-58 of
        Durbin et al.

        Returns:
         - A dictionary containing the forward variables. This has keys of the
           form (state letter, position in the training sequence), and values
           containing the calculated forward variable.
         - The calculated probability of the sequence.

        """
    state_letters = self._mm.state_alphabet
    forward_var = {}
    forward_var[state_letters[0], -1] = 1
    for k in range(1, len(state_letters)):
        forward_var[state_letters[k], -1] = 0
    for i in range(len(self._seq.emissions)):
        for main_state in state_letters:
            forward_value = self._forward_recursion(main_state, i, forward_var)
            if forward_value is not None:
                forward_var[main_state, i] = forward_value
    first_state = state_letters[0]
    seq_prob = 0
    for state_item in state_letters:
        forward_value = forward_var[state_item, len(self._seq.emissions) - 1]
        transition_value = self._mm.transition_prob[state_item, first_state]
        seq_prob += forward_value * transition_value
    return (forward_var, seq_prob)