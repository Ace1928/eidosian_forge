import _lsprof
import io
import profile as _pyprofile
def snapshot_stats(self):
    entries = self.getstats()
    self.stats = {}
    callersdicts = {}
    for entry in entries:
        func = label(entry.code)
        nc = entry.callcount
        cc = nc - entry.reccallcount
        tt = entry.inlinetime
        ct = entry.totaltime
        callers = {}
        callersdicts[id(entry.code)] = callers
        self.stats[func] = (cc, nc, tt, ct, callers)
    for entry in entries:
        if entry.calls:
            func = label(entry.code)
            for subentry in entry.calls:
                try:
                    callers = callersdicts[id(subentry.code)]
                except KeyError:
                    continue
                nc = subentry.callcount
                cc = nc - subentry.reccallcount
                tt = subentry.inlinetime
                ct = subentry.totaltime
                if func in callers:
                    prev = callers[func]
                    nc += prev[0]
                    cc += prev[1]
                    tt += prev[2]
                    ct += prev[3]
                callers[func] = (nc, cc, tt, ct)