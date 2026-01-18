import sys
from enchant.checker import SpellChecker
def print_suggestions(self):
    """Prints out the suggestions for a given error.

        This function will add vertical pipes to separate choices
        as well as the index of the replacement as expected by the replace function.
        I don't believe zero indexing is a problem as long as the user can see the numbers :)
        """
    result = ''
    suggestions = self.error.suggest()
    for index, sugg in enumerate(suggestions):
        if index == 0:
            result = result + color(str(index), color='yellow') + ': ' + color(sugg, color='bold')
        else:
            result = result + ' | ' + color(str(index), color='yellow') + ': ' + color(sugg, color='bold')
    print([info('HOW ABOUT:'), result])