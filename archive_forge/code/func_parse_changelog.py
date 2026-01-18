import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def parse_changelog(self, file, max_blocks=None, allow_empty_author=False, strict=True, encoding=None):
    """ Read and parse a changelog file

        If you create an Changelog object without specifying a changelog
        file, you can parse a changelog file with this method. If the
        changelog doesn't parse cleanly, a :class:`ChangelogParseError`
        exception is thrown. The constructor will parse the changelog on
        a best effort basis.
        """
    first_heading = 'first heading'
    next_heading_or_eof = 'next heading of EOF'
    start_of_change_data = 'start of change data'
    more_changes_or_trailer = 'more change data or trailer'
    slurp_to_end = 'slurp to end'
    encoding = encoding or self._encoding
    if file is None:
        self._parse_error('Empty changelog file.', strict)
        return
    self._blocks = []
    self.initial_blank_lines = []
    current_block = ChangeBlock(encoding=encoding)
    changes = []
    state = first_heading
    old_state = None
    if isinstance(file, bytes):
        file = file.decode(encoding)
    if isinstance(file, str):
        if not file.strip():
            self._parse_error('Empty changelog file.', strict)
            return
        file = file.splitlines()
    for line in file:
        if not isinstance(line, str):
            line = line.decode(encoding)
        line = line.rstrip('\n')
        if state in (first_heading, next_heading_or_eof):
            top_match = topline.match(line)
            blank_match = blankline.match(line)
            if top_match is not None:
                if max_blocks is not None and len(self._blocks) >= max_blocks:
                    return
                current_block.package = top_match.group(1)
                current_block._raw_version = top_match.group(2)
                current_block.distributions = top_match.group(3).lstrip()
                pairs = line.split(';', 1)[1]
                all_keys = {}
                other_pairs = {}
                for pair in pairs.split(','):
                    pair = pair.strip()
                    kv_match = keyvalue.match(pair)
                    if kv_match is None:
                        self._parse_error("Invalid key-value pair after ';': %s" % pair, strict)
                        continue
                    key = kv_match.group(1)
                    value = kv_match.group(2)
                    if key.lower() in all_keys:
                        self._parse_error('Repeated key-value: %s' % key.lower(), strict)
                    all_keys[key.lower()] = value
                    if key.lower() == 'urgency':
                        val_match = value_re.match(value)
                        if val_match is None:
                            self._parse_error('Badly formatted urgency value: %s' % value, strict)
                        else:
                            current_block.urgency = val_match.group(1)
                            comment = val_match.group(2)
                            if comment is not None:
                                current_block.urgency_comment = comment
                    else:
                        other_pairs[key] = value
                current_block.other_pairs = other_pairs
                state = start_of_change_data
            elif blank_match is not None:
                if state == first_heading:
                    self.initial_blank_lines.append(line)
                else:
                    self._blocks[-1].add_trailing_line(line)
            else:
                emacs_match = emacs_variables.match(line)
                vim_match = vim_variables.match(line)
                cvs_match = cvs_keyword.match(line)
                comments_match = comments.match(line)
                more_comments_match = more_comments.match(line)
                if (emacs_match is not None or vim_match is not None) and state != first_heading:
                    self._blocks[-1].add_trailing_line(line)
                    old_state = state
                    state = slurp_to_end
                    continue
                if cvs_match is not None or comments_match is not None or more_comments_match is not None:
                    if state == first_heading:
                        self.initial_blank_lines.append(line)
                    else:
                        self._blocks[-1].add_trailing_line(line)
                    continue
                if (old_format_re1.match(line) is not None or old_format_re2.match(line) is not None or old_format_re3.match(line) is not None or (old_format_re4.match(line) is not None) or (old_format_re5.match(line) is not None) or (old_format_re6.match(line) is not None) or (old_format_re7.match(line) is not None) or (old_format_re8.match(line) is not None)) and state != first_heading:
                    self._blocks[-1].add_trailing_line(line)
                    old_state = state
                    state = slurp_to_end
                    continue
                self._parse_error('Unexpected line while looking for %s: %s' % (state, line), strict)
                if state == first_heading:
                    self.initial_blank_lines.append(line)
                else:
                    self._blocks[-1].add_trailing_line(line)
        elif state in (start_of_change_data, more_changes_or_trailer):
            change_match = changere.match(line)
            end_match = endline.match(line)
            end_no_details_match = endline_nodetails.match(line)
            blank_match = blankline.match(line)
            if change_match is not None:
                changes.append(line)
                state = more_changes_or_trailer
            elif end_match is not None:
                if end_match.group(3) != '  ':
                    self._parse_error('Badly formatted trailer line: %s' % line, strict)
                    current_block._trailer_separator = end_match.group(3)
                current_block.author = '%s <%s>' % (end_match.group(1), end_match.group(2))
                current_block.date = end_match.group(4)
                current_block._changes = changes
                self._blocks.append(current_block)
                changes = []
                current_block = ChangeBlock(encoding=encoding)
                state = next_heading_or_eof
            elif end_no_details_match is not None:
                if not allow_empty_author:
                    self._parse_error('Badly formatted trailer line: %s' % line, strict)
                    continue
                current_block._changes = changes
                self._blocks.append(current_block)
                changes = []
                current_block = ChangeBlock(encoding=encoding)
                state = next_heading_or_eof
            elif blank_match is not None:
                changes.append(line)
            else:
                cvs_match = cvs_keyword.match(line)
                comments_match = comments.match(line)
                more_comments_match = more_comments.match(line)
                if cvs_match is not None or comments_match is not None or more_comments_match is not None:
                    changes.append(line)
                    continue
                self._parse_error('Unexpected line while looking for %s: %s' % (state, line), strict)
                changes.append(line)
        elif state == slurp_to_end:
            if old_state == next_heading_or_eof:
                self._blocks[-1].add_trailing_line(line)
            else:
                changes.append(line)
        else:
            assert False, 'Unknown state: %s' % state
    if state not in (next_heading_or_eof, slurp_to_end) or (state == slurp_to_end and old_state != next_heading_or_eof):
        self._parse_error('Found eof where expected %s' % state, strict)
        current_block._changes = changes
        current_block._no_trailer = True
        self._blocks.append(current_block)