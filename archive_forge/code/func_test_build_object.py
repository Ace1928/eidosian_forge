import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.tests.compat import mock
def test_build_object(self):
    datetest = (DateTuple('1', '2', '3', '4', '5', '6'), {'YYYY': '1', 'MM': '2', 'DD': '3', 'Www': '4', 'D': '5', 'DDD': '6'})
    timetest = (TimeTuple('1', '2', '3', TimezoneTuple(False, False, '4', '5', 'tz name')), {'hh': '1', 'mm': '2', 'ss': '3', 'tz': TimezoneTuple(False, False, '4', '5', 'tz name')})
    datetimetest = (DatetimeTuple(DateTuple('1', '2', '3', '4', '5', '6'), TimeTuple('7', '8', '9', TimezoneTuple(True, False, '10', '11', 'tz name'))), (DateTuple('1', '2', '3', '4', '5', '6'), TimeTuple('7', '8', '9', TimezoneTuple(True, False, '10', '11', 'tz name'))))
    durationtest = (DurationTuple('1', '2', '3', '4', '5', '6', '7'), {'PnY': '1', 'PnM': '2', 'PnW': '3', 'PnD': '4', 'TnH': '5', 'TnM': '6', 'TnS': '7'})
    intervaltests = ((IntervalTuple(DateTuple('1', '2', '3', '4', '5', '6'), DateTuple('7', '8', '9', '10', '11', '12'), None), {'start': DateTuple('1', '2', '3', '4', '5', '6'), 'end': DateTuple('7', '8', '9', '10', '11', '12'), 'duration': None}), (IntervalTuple(DateTuple('1', '2', '3', '4', '5', '6'), None, DurationTuple('7', '8', '9', '10', '11', '12', '13')), {'start': DateTuple('1', '2', '3', '4', '5', '6'), 'end': None, 'duration': DurationTuple('7', '8', '9', '10', '11', '12', '13')}), (IntervalTuple(None, TimeTuple('1', '2', '3', TimezoneTuple(True, False, '4', '5', 'tz name')), DurationTuple('6', '7', '8', '9', '10', '11', '12')), {'start': None, 'end': TimeTuple('1', '2', '3', TimezoneTuple(True, False, '4', '5', 'tz name')), 'duration': DurationTuple('6', '7', '8', '9', '10', '11', '12')}))
    repeatingintervaltests = ((RepeatingIntervalTuple(True, None, IntervalTuple(DateTuple('1', '2', '3', '4', '5', '6'), DateTuple('7', '8', '9', '10', '11', '12'), None)), {'R': True, 'Rnn': None, 'interval': IntervalTuple(DateTuple('1', '2', '3', '4', '5', '6'), DateTuple('7', '8', '9', '10', '11', '12'), None)}), (RepeatingIntervalTuple(False, '1', IntervalTuple(DatetimeTuple(DateTuple('2', '3', '4', '5', '6', '7'), TimeTuple('8', '9', '10', None)), DatetimeTuple(DateTuple('11', '12', '13', '14', '15', '16'), TimeTuple('17', '18', '19', None)), None)), {'R': False, 'Rnn': '1', 'interval': IntervalTuple(DatetimeTuple(DateTuple('2', '3', '4', '5', '6', '7'), TimeTuple('8', '9', '10', None)), DatetimeTuple(DateTuple('11', '12', '13', '14', '15', '16'), TimeTuple('17', '18', '19', None)), None)}))
    timezonetest = (TimezoneTuple(False, False, '1', '2', '+01:02'), {'negative': False, 'Z': False, 'hh': '1', 'mm': '2', 'name': '+01:02'})
    with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_date') as mock_build:
        mock_build.return_value = datetest[0]
        result = BaseTimeBuilder._build_object(datetest[0])
        self.assertEqual(result, datetest[0])
        mock_build.assert_called_once_with(**datetest[1])
    with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_time') as mock_build:
        mock_build.return_value = timetest[0]
        result = BaseTimeBuilder._build_object(timetest[0])
        self.assertEqual(result, timetest[0])
        mock_build.assert_called_once_with(**timetest[1])
    with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_datetime') as mock_build:
        mock_build.return_value = datetimetest[0]
        result = BaseTimeBuilder._build_object(datetimetest[0])
        self.assertEqual(result, datetimetest[0])
        mock_build.assert_called_once_with(*datetimetest[1])
    with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_duration') as mock_build:
        mock_build.return_value = durationtest[0]
        result = BaseTimeBuilder._build_object(durationtest[0])
        self.assertEqual(result, durationtest[0])
        mock_build.assert_called_once_with(**durationtest[1])
    for intervaltest in intervaltests:
        with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_interval') as mock_build:
            mock_build.return_value = intervaltest[0]
            result = BaseTimeBuilder._build_object(intervaltest[0])
            self.assertEqual(result, intervaltest[0])
            mock_build.assert_called_once_with(**intervaltest[1])
    for repeatingintervaltest in repeatingintervaltests:
        with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_repeating_interval') as mock_build:
            mock_build.return_value = repeatingintervaltest[0]
            result = BaseTimeBuilder._build_object(repeatingintervaltest[0])
            self.assertEqual(result, repeatingintervaltest[0])
            mock_build.assert_called_once_with(**repeatingintervaltest[1])
    with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_timezone') as mock_build:
        mock_build.return_value = timezonetest[0]
        result = BaseTimeBuilder._build_object(timezonetest[0])
        self.assertEqual(result, timezonetest[0])
        mock_build.assert_called_once_with(**timezonetest[1])