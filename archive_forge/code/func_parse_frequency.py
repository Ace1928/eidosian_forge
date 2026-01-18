import datetime
import re
def parse_frequency(frequency):
    frequencies = {'hourly': Frequencies.hourly, 'daily': Frequencies.daily, 'weekly': Frequencies.weekly, 'monthly': Frequencies.monthly, 'yearly': Frequencies.yearly}
    frequency = frequency.strip().lower()
    return frequencies.get(frequency, None)