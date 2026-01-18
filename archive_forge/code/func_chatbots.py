from nltk.chat.eliza import eliza_chat
from nltk.chat.iesha import iesha_chat
from nltk.chat.rude import rude_chat
from nltk.chat.suntsu import suntsu_chat
from nltk.chat.util import Chat
from nltk.chat.zen import zen_chat
def chatbots():
    print('Which chatbot would you like to talk to?')
    botcount = len(bots)
    for i in range(botcount):
        print('  %d: %s' % (i + 1, bots[i][1]))
    while True:
        choice = input(f'\nEnter a number in the range 1-{botcount}: ').strip()
        if choice.isdigit() and int(choice) - 1 in range(botcount):
            break
        else:
            print('   Error: bad chatbot number')
    chatbot = bots[int(choice) - 1][0]
    chatbot()