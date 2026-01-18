import contextlib
import sys
import unittest
from io import StringIO
from nltk.corpus import gutenberg
from nltk.text import Text
def test_concordance_print(self):
    print_out = 'Displaying 11 of 11 matches:\n        ong the former , one was of a most monstrous size . ... This came towards us ,\n        ON OF THE PSALMS . " Touching that monstrous bulk of the whale or ork we have r\n        ll over with a heathenish array of monstrous clubs and spears . Some were thick\n        d as you gazed , and wondered what monstrous cannibal and savage could ever hav\n        that has survived the flood ; most monstrous and most mountainous ! That Himmal\n        they might scout at Moby Dick as a monstrous fable , or still worse and more de\n        th of Radney .\'" CHAPTER 55 Of the Monstrous Pictures of Whales . I shall ere l\n        ing Scenes . In connexion with the monstrous pictures of whales , I am strongly\n        ere to enter upon those still more monstrous stories of them which are to be fo\n        ght have been rummaged out of this monstrous cabinet there is no telling . But\n        of Whale - Bones ; for Whales of a monstrous size are oftentimes cast up dead u\n        '
    with stdout_redirect(StringIO()) as stdout:
        self.text.concordance(self.query)

    def strip_space(raw_str):
        return raw_str.replace(' ', '')
    self.assertEqual(strip_space(print_out), strip_space(stdout.getvalue()))