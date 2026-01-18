import curses
import curses.ascii
def test_editbox(stdscr):
    ncols, nlines = (9, 4)
    uly, ulx = (15, 20)
    stdscr.addstr(uly - 2, ulx, 'Use Ctrl-G to end editing.')
    win = curses.newwin(nlines, ncols, uly, ulx)
    rectangle(stdscr, uly - 1, ulx - 1, uly + nlines, ulx + ncols)
    stdscr.refresh()
    return Textbox(win).edit()