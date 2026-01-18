def count_bads(self, recorded_events):
    """returns counts of anomalies as a dict of anomaly: count

        recorded_events: media_player events recorded while playing a sample

        Notice that 'counters' has one more key than 'bads': "scheduling_in_past"
        """
    recorded_events = self.preprocessing(recorded_events)
    counters = {k: 0 for k in self.bads}
    cnt_scheduling_in_past = 0
    mp_states = MediaPlayerStateIterator(recorded_events, self.events_definition)
    for st in mp_states:
        evname = st['evname']
        if evname in counters:
            counters[evname] += 1
        elif 'p.P.ut.1.9' and st['rescheduling_time'] is not None and (st['rescheduling_time'] < 0):
            cnt_scheduling_in_past += 1
    counters['scheduling_in_past'] = cnt_scheduling_in_past
    return counters