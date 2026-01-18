import collections
import functools
from taskflow import states
from taskflow import test
from taskflow.types import notifier as nt
def test_different_details_filter(self):
    call_counts = collections.defaultdict(list)

    def call_me_on(registered_state, state, details):
        call_counts[registered_state].append((state, details))

    def when_red(details):
        return details.get('color') == 'red'

    def when_blue(details):
        return details.get('color') == 'blue'
    notifier = nt.Notifier()
    call_me_on_success = functools.partial(call_me_on, states.SUCCESS)
    notifier.register(states.SUCCESS, call_me_on_success, details_filter=when_red)
    notifier.register(states.SUCCESS, call_me_on_success, details_filter=when_blue)
    self.assertEqual(2, len(notifier))
    self.assertTrue(notifier.is_registered(states.SUCCESS, call_me_on_success, details_filter=when_blue))
    self.assertTrue(notifier.is_registered(states.SUCCESS, call_me_on_success, details_filter=when_red))
    notifier.notify(states.SUCCESS, {})
    self.assertEqual(0, len(call_counts[states.SUCCESS]))
    notifier.notify(states.SUCCESS, {'color': 'red'})
    self.assertEqual(1, len(call_counts[states.SUCCESS]))
    notifier.notify(states.SUCCESS, {'color': 'blue'})
    self.assertEqual(2, len(call_counts[states.SUCCESS]))
    notifier.notify(states.SUCCESS, {'color': 'green'})
    self.assertEqual(2, len(call_counts[states.SUCCESS]))